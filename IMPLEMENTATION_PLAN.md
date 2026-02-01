# Food Tracker Mobile App - Implementation Plan

## Project Overview
A frictionless macro/food tracking mobile app with AI-powered batch photo processing, scale-aware estimation, and persistent photo storage.

## Tech Stack

### Mobile App (React Native + Expo)
- **Framework**: React Native 0.81.5, Expo ~54.0.32
- **Language**: TypeScript 5.9.2
- **Navigation**: React Navigation (native stack + bottom tabs)
- **State Management**: Zustand 5.0.0
- **UI/Animations**: React Native Reanimated, Gesture Handler, Gorhom Bottom Sheet
- **Storage**: AsyncStorage (local preferences), expo-file-system
- **Camera/Media**: expo-image-picker, expo-media-library
- **Testing**: Jest, React Testing Library

### AI/Backend Services
- **AI Framework**: Google Agent Development Kit (ADK)
- **Vision Models**: Google Gemini with vision capabilities
- **Storage**: Google Cloud Storage (GCS) for photos
- **Database**: PostgreSQL/Supabase for user data, logs, and entry history
- **APIs**: RESTful backend (Node.js/Express or Python/FastAPI)

### Food Composition Databases
- **Australia**: AFCD/NUTTAB
- **North America**: USDA FoodData Central, Health Canada CNF
- **Europe**: CoFID (UK), CIQUAL (France), EuroFIR
- **Asia**: STFCJ (Japan), China Food Composition Tables, ASEANFOODS
- **Global Fallback**: Open Food Facts

## Project Structure

```
foodtracker/
├── apps/
│   └── mobile/                    # React Native app
│       ├── src/
│       │   ├── screens/          # Screen components
│       │   │   ├── HomeScreen.tsx
│       │   │   ├── PhotoBatchScreen.tsx
│       │   │   ├── DiaryScreen.tsx
│       │   │   ├── EntryDetailScreen.tsx
│       │   │   └── ProfileScreen.tsx
│       │   ├── components/       # Reusable components
│       │   │   ├── PhotoPicker.tsx
│       │   │   ├── BatchPhotoGrid.tsx
│       │   │   ├── FoodEntryCard.tsx
│       │   │   ├── IngredientList.tsx
│       │   │   ├── ScaleDetectionIndicator.tsx
│       │   │   └── NutritionSummary.tsx
│       │   ├── navigation/       # Navigation setup
│       │   │   ├── RootNavigator.tsx
│       │   │   ├── MainTabNavigator.tsx
│       │   │   └── types.ts
│       │   ├── store/           # Zustand stores
│       │   │   ├── useAuthStore.ts
│       │   │   ├── useFoodLogStore.ts
│       │   │   ├── usePhotoStore.ts
│       │   │   └── usePreferencesStore.ts
│       │   ├── lib/             # Utilities and API clients
│       │   │   ├── api/
│       │   │   │   ├── client.ts
│       │   │   │   ├── authApi.ts
│       │   │   │   ├── foodLogApi.ts
│       │   │   │   ├── photoApi.ts
│       │   │   │   └── nutritionApi.ts
│       │   │   ├── imageProcessing.ts
│       │   │   └── nutritionCalculations.ts
│       │   └── types/           # TypeScript types
│       │       └── index.ts
│       ├── package.json
│       ├── app.json
│       └── tsconfig.json
├── services/
│   └── ai-agent/                # Google ADK agents
│       ├── src/
│       │   ├── agents/
│       │   │   ├── coordinator.ts        # Root orchestrator
│       │   │   ├── visionAgent.ts        # Image segmentation
│       │   │   ├── scaleAgent.ts         # OCR for scale weight
│       │   │   ├── volumetricAgent.ts    # Volume estimation
│       │   │   ├── databaseAgent.ts      # DB routing
│       │   │   └── branchingAgent.ts     # Hypothesis testing
│       │   ├── tools/
│       │   │   ├── imageSegmentation.ts
│       │   │   ├── ocrTool.ts
│       │   │   └── densityLookup.ts
│       │   ├── config/
│       │   │   └── databases.ts         # FCDB configs
│       │   └── index.ts
│       └── package.json
└── backend/                     # API server
    ├── src/
    │   ├── routes/
    │   │   ├── auth.ts
    │   │   ├── photos.ts
    │   │   ├── foodLogs.ts
    │   │   └── nutrition.ts
    │   ├── services/
    │   │   ├── gcsService.ts           # Photo storage
    │   │   ├── adkService.ts           # ADK agent calls
    │   │   └── nutritionDbService.ts   # FCDB queries
    │   ├── db/
    │   │   ├── schema.sql
    │   │   └── migrations/
    │   └── index.ts
    └── package.json
```

## Implementation Phases

### Phase 1: Project Setup & Foundation (Week 1)

#### 1.1 Initialize Mobile App
- [ ] Create Expo project with TypeScript template
- [ ] Install core dependencies (React Navigation, Zustand, Reanimated)
- [ ] Set up folder structure matching router-builder pattern
- [ ] Configure ESLint, Prettier, TypeScript
- [ ] Set up basic navigation (tabs: Home, Diary, Profile)
- [ ] Create placeholder screens

#### 1.2 Backend & Database Setup
- [ ] Initialize Node.js/Express backend (or FastAPI)
- [ ] Set up PostgreSQL database (local + Supabase cloud)
- [ ] Create database schema:
  - `users` table
  - `food_entries` table (with photo_urls JSON array)
  - `ingredients` table (linked to entries, with modification_history)
  - `photos` table (GCS URLs, metadata)
- [ ] Set up Google Cloud Storage bucket for photos
- [ ] Implement basic auth endpoints (signup, login, JWT)

#### 1.3 AI Agent Infrastructure
- [ ] Set up Google ADK project
- [ ] Configure Gemini API access
- [ ] Create basic CoordinatorAgent structure
- [ ] Test ADK connection with simple prompt

### Phase 2: Core Photo Management (Week 2)

#### 2.1 Photo Selection & Batch Upload
- [ ] Implement PhotoPicker component using expo-image-picker
- [ ] Support multi-select (up to 20 images)
- [ ] Create BatchPhotoGrid component to display selected photos
- [ ] Add local caching of photos before upload
- [ ] Implement upload to GCS with progress indicators
- [ ] Handle upload errors and retry logic

#### 2.2 Photo Storage Service
- [ ] Create GCS service for signed URL generation
- [ ] Implement photo compression before upload (maintain EXIF)
- [ ] Store photo metadata (timestamp, location if available, dimensions)
- [ ] Link photos to food entries in database
- [ ] Implement photo retrieval for entry detail view

### Phase 3: AI Vision Processing (Week 3-4)

#### 3.1 Vision Specialist Agent (ADK)
- [ ] Implement VisionAgent for food segmentation
- [ ] Use Gemini vision model for ingredient identification
- [ ] Extract bounding boxes for each detected food item
- [ ] Identify container type (plate, bowl, scale)
- [ ] Return structured JSON with detected items

#### 3.2 Scale Analysis Agent
- [ ] Implement OCR detection for digital scale displays
- [ ] Extract weight value and unit (g, kg, oz, lb)
- [ ] Detect analog scale readings (if feasible)
- [ ] Flag images as "scale present" or "no scale"
- [ ] Store scale metadata with photo

#### 3.3 Volumetric Estimation Agent
- [ ] Implement 3D volume estimation from 2D images
- [ ] Use reference objects (coin, standard plate size) for calibration
- [ ] Calculate pixel-to-mm ratio
- [ ] Estimate volume for each segmented food item
- [ ] Account for container depth and food packing density

### Phase 4: Hypothesis Branching & Database Routing (Week 5)

#### 4.1 Branching Agent Logic
- [ ] Implement hypothesis generation for scale weights:
  - Branch A: Tared weight (Wfood = Wscale)
  - Branch B: Standard container (Wfood = Wscale - Wplate_standard)
  - Branch C: Custom container (Wfood = Wscale - Vcontainer × ρcontainer)
- [ ] Calculate error for each branch using food density models
- [ ] Select best-fit branch (lowest error)
- [ ] Store all branches in session.state for user override

#### 4.2 Database Routing Agent
- [ ] Implement geolocation-based database selection:
  - Australia → AFCD/NUTTAB
  - USA → USDA FoodData Central
  - Canada → Health Canada CNF
  - UK → CoFID
  - France → CIQUAL
  - Global fallback → Open Food Facts
- [ ] Create unified nutrition data schema
- [ ] Implement fuzzy matching for ingredient names
- [ ] Handle branded foods with barcode support

#### 4.3 Food Composition Database Integration
- [ ] Set up USDA FoodData Central API client
- [ ] Download and cache AFCD/NUTTAB datasets
- [ ] Integrate Open Food Facts API
- [ ] Create density lookup tables for common foods
- [ ] Implement micronutrient calculation

### Phase 5: Entry Management & Retrospective Editing (Week 6)

#### 5.1 Food Entry Creation
- [ ] Create FoodEntryCard component
- [ ] Display ingredients list with quantities
- [ ] Show nutrition summary (calories, protein, carbs, fat)
- [ ] Link to original photos (persistent storage)
- [ ] Implement entry creation from AI results

#### 5.2 Retrospective Editing
- [ ] Create EntryDetailScreen with editable ingredients
- [ ] Allow user to modify ingredient quantities
- [ ] Add/remove ingredients from existing entries
- [ ] Re-run weight validation when ingredients change
- [ ] Show modification history
- [ ] Visual indicator if entry deviates from AI estimate

#### 5.3 Diary View
- [ ] Implement DiaryScreen with daily log view
- [ ] Show timeline of entries with thumbnails
- [ ] Display daily nutrition totals
- [ ] Calendar view for historical entries
- [ ] Search/filter functionality

### Phase 6: AI Agent Orchestration (Week 7)

#### 6.1 Multi-Agent System (ADK)
- [ ] Implement ParallelAgent for batch photo processing
- [ ] Create SequentialAgent pipeline:
  1. Vision detection
  2. Volume estimation
  3. Scale analysis
  4. Branching logic
  5. Database routing
  6. Ingredient compilation
- [ ] Use session.state for shared context across agents
- [ ] Implement LoopAgent for iterative refinement

#### 6.2 Agent Evaluation & Testing
- [ ] Set up evaluation dataset (Nutrition5k, Uni-Food)
- [ ] Measure Mean Absolute Error (MAE) for weight estimation
- [ ] Calculate caloric accuracy vs ground truth
- [ ] Test branching logic accuracy on scale images
- [ ] Optimize agent prompts based on eval results

### Phase 7: UI/UX Polish (Week 8)

#### 7.1 Animations & Interactions
- [ ] Add smooth transitions between screens
- [ ] Implement photo upload progress animations
- [ ] Create loading states for AI processing
- [ ] Add haptic feedback for key actions
- [ ] Implement pull-to-refresh on diary

#### 7.2 User Preferences
- [ ] Profile screen with user settings
- [ ] Nutrition goals (calories, macros)
- [ ] Regional database selection
- [ ] Unit preferences (metric/imperial)
- [ ] Dark mode support

#### 7.3 Error Handling
- [ ] Graceful degradation if AI fails
- [ ] Manual entry fallback
- [ ] Photo re-upload option
- [ ] Network error handling
- [ ] Offline mode with sync queue

### Phase 8: Monetization & Advanced Features (Week 9-10)

#### 8.1 Subscription System
- [ ] Integrate RevenueCat or Stripe
- [ ] Implement freemium tier:
  - Single photo logs
  - 7-day history
  - Barcode scanning
- [ ] Pro tier ($12.99/month, $64.99/year):
  - Batch uploads
  - Persistent photos
  - Scale-weight AI
  - Unlimited history
- [ ] Family plan ($19.99/month, $89.99/year)

#### 8.2 Advanced Features
- [ ] Barcode scanner for packaged foods
- [ ] Recipe builder
- [ ] Meal planning
- [ ] Export to CSV/PDF
- [ ] Integration with fitness apps (Apple Health, Google Fit)

### Phase 9: Testing & Quality Assurance (Week 11)

#### 9.1 Unit Tests
- [ ] Test React Native components
- [ ] Test Zustand stores
- [ ] Test API clients
- [ ] Test utility functions

#### 9.2 Integration Tests
- [ ] Test photo upload flow
- [ ] Test AI agent pipeline
- [ ] Test entry creation and editing
- [ ] Test database queries

#### 9.3 End-to-End Tests
- [ ] Test complete user journey (photo → entry)
- [ ] Test batch processing
- [ ] Test scale weight scenarios
- [ ] Test offline/online transitions

### Phase 10: Deployment & Launch (Week 12)

#### 10.1 Backend Deployment
- [ ] Deploy to Google Cloud Run or Railway
- [ ] Set up production database
- [ ] Configure CDN for photo delivery
- [ ] Set up monitoring (Sentry, LogRocket)

#### 10.2 Mobile App Deployment
- [ ] Build production iOS app
- [ ] Build production Android app
- [ ] Submit to App Store
- [ ] Submit to Google Play
- [ ] Set up analytics (Mixpanel, Amplitude)

#### 10.3 Marketing & SEO
- [ ] Create landing page
- [ ] Optimize for keywords:
  - "Agentic Calorie Tracker"
  - "Batch Food Photo App"
  - "AI Weight Estimation Food Scale"
  - "Australian Food Database App"
- [ ] Blog content on nutrition tracking
- [ ] Social media presence

## Key Architectural Decisions

### Why Google ADK?
- **Modularity**: Each agent has a single responsibility
- **Scalability**: Parallel processing of batch photos
- **Maintainability**: Easy to update individual agents
- **Orchestration**: Built-in support for sequential, parallel, and loop patterns

### Why Zustand for State Management?
- Minimal boilerplate compared to Redux
- TypeScript-friendly
- Supports middleware (persist, devtools)
- Small bundle size

### Why Expo Managed Workflow?
- Faster development with OTA updates
- Built-in camera, file system, media library
- Easy deployment with EAS Build
- Good performance with modern architecture

### Database Schema Highlights

```sql
-- Food entries with persistent photo links
CREATE TABLE food_entries (
  id UUID PRIMARY KEY,
  user_id UUID REFERENCES users(id),
  created_at TIMESTAMP DEFAULT NOW(),
  meal_type TEXT, -- breakfast, lunch, dinner, snack
  photos JSONB, -- Array of GCS URLs with metadata
  total_calories FLOAT,
  total_protein FLOAT,
  total_carbs FLOAT,
  total_fat FLOAT,
  modification_history JSONB -- Track user edits
);

-- Individual ingredients (many-to-one with entries)
CREATE TABLE ingredients (
  id UUID PRIMARY KEY,
  entry_id UUID REFERENCES food_entries(id),
  name TEXT,
  quantity FLOAT,
  unit TEXT,
  calories FLOAT,
  protein FLOAT,
  carbs FLOAT,
  fat FLOAT,
  source_segment JSONB, -- Which pixels in photo
  ai_confidence FLOAT,
  user_modified BOOLEAN DEFAULT FALSE,
  database_source TEXT -- AFCD, USDA, etc.
);
```

## Success Metrics

### Technical Metrics
- **AI Accuracy**: MAE < 15% for weight estimation
- **Caloric Accuracy**: Within 10% of ground truth
- **Processing Speed**: < 5 seconds per photo in batch
- **App Performance**: 60fps animations, < 2s screen transitions

### Business Metrics
- **User Retention**: 40%+ after 30 days
- **Trial Conversion**: 8-10% freemium → Pro
- **Daily Active Usage**: 70%+ of registered users
- **Logging Friction**: < 2 minutes per day average

## Research Tasks (Using Gemini CLI)

These tasks should be offloaded to Gemini for large-scale analysis:

1. **Dataset Analysis**
   ```bash
   gemini -p "Research Nutrition5k dataset. Provide download links, data format,
   and sample code for loading images and ground truth weights."
   ```

2. **FCDB API Documentation**
   ```bash
   gemini -p "Analyze USDA FoodData Central API documentation. List all endpoints,
   authentication methods, rate limits, and provide sample queries for searching
   foods by name and getting nutrition data."
   ```

3. **Competitor Feature Analysis**
   ```bash
   gemini -p "Research MacroFactor, MyFitnessPal, and Cronometer apps. List their
   photo logging features, editability after AI detection, and any scale-aware
   estimation features. Create comparison table."
   ```

4. **ADK Architecture Patterns**
   ```bash
   gemini -p "@services/ai-agent/ Analyze this ADK agent structure. Suggest
   improvements for the branching logic and session state management."
   ```

## Next Steps

1. **Create Expo project** with TypeScript
2. **Set up basic navigation** structure
3. **Implement photo picker** with multi-select
4. **Create simple ADK agent** to test Gemini vision
5. **Build basic food entry UI**
6. **Integrate USDA API** as first database

Let's start building! 🚀
