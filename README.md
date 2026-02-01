# Food Tracker - AI-Powered Macro Tracking App

A frictionless food tracking mobile app with AI-powered batch photo processing, scale-aware estimation, and persistent photo storage.

## Project Structure

```
foodtracker/
├── apps/
│   └── mobile/              # React Native + Expo mobile app
│       ├── src/
│       │   ├── screens/     # App screens (Home, Diary, Profile)
│       │   ├── components/  # PhotoPicker, BatchPhotoGrid
│       │   ├── navigation/  # React Navigation setup
│       │   ├── store/       # Zustand state management
│       │   ├── lib/api/     # API clients for backend
│       │   └── types/       # TypeScript definitions
│       └── package.json
├── backend/                 # Express API server
│   ├── src/
│   │   ├── routes/         # API routes (food-logs)
│   │   └── services/       # AI service integration
│   └── package.json
├── services/
│   └── ai-agent/            # Google ADK AI agent service
│       ├── src/
│       │   ├── agents/      # Vision, Scale, Coordinator agents
│       │   ├── config/      # Database configs, food density tables
│       │   └── tools/       # AI agent tools (future)
│       └── package.json
├── context.md               # Project goals and research
├── IMPLEMENTATION_PLAN.md   # Detailed implementation roadmap
└── README.md               # This file
```

## Tech Stack

### Mobile App
- **Framework**: React Native 0.81.5 + Expo ~54.0.32
- **Language**: TypeScript 5.9.2
- **Navigation**: React Navigation (Stack + Tabs)
- **State Management**: Zustand 5.0.0
- **UI**: React Native Reanimated, Gesture Handler

### Backend API
- **Framework**: Express.js 4.18
- **Language**: TypeScript (ES Modules)
- **Features**: Photo processing, food log management

### AI Service
- **Framework**: Google Agent Development Kit (ADK)
- **Model**: Gemini 2.0 Flash
- **Language**: TypeScript (ES Modules)

## Key Features (Planned)

1. **Batch Photo Upload**: Upload multiple food photos at once to reduce friction
2. **Persistent Photos**: Photos saved permanently and linked to entries
3. **Retrospective Editing**: Modify ingredients days after initial entry
4. **Scale-Aware AI**: Detects scale weights and uses branching logic for accuracy
5. **Multi-Agent Processing**: Vision, OCR, volumetric estimation, hypothesis testing
6. **Global Database Support**: AFCD (AU), USDA (US), CoFID (UK), CIQUAL (FR), etc.

## Current Status

### ✅ Completed (Phase 1 & 2)
- [x] Mobile app initialization with Expo + TypeScript
- [x] React Navigation setup (Tab + Stack navigators)
- [x] Home, Diary, Profile screens
- [x] Zustand stores for food logs and preferences
- [x] TypeScript type definitions
- [x] **PhotoPicker component** - Multi-select up to 20 photos
- [x] **BatchPhotoGrid component** - Display selected photos
- [x] **API client** - HTTP client for backend communication
- [x] **Express backend** - API server with food-logs endpoints
- [x] AI agent service structure with Google ADK
- [x] Vision Agent (food detection)
- [x] Scale Agent (weight reading)
- [x] Coordinator Agent (orchestrator)
- [x] Food composition database configurations

### 🚧 In Progress (Phase 3)
- [ ] Connect Vision Agent to actual Gemini API
- [ ] Implement photo upload to cloud storage
- [ ] Database setup (PostgreSQL/Supabase)

## Getting Started

### Mobile App

```bash
cd apps/mobile
npm install
npm run ios     # or npm run android
```

### Backend API

```bash
cd backend
npm install
npm run dev     # Starts on http://localhost:3000
```

### AI Agent Service

```bash
cd services/ai-agent

# Set up environment
cp .env.example .env
# Add your GOOGLE_API_KEY to .env

npm install
npm run build
npm run dev
```

### Running the Full Stack

Open 3 terminal windows:

```bash
# Terminal 1: Backend API
cd backend && npm run dev

# Terminal 2: Mobile App
cd apps/mobile && npm start

# Terminal 3: AI Agent Service (when needed)
cd services/ai-agent && npm run dev
```

## Development Workflow

1. **Using Gemini CLI for Research**:
   ```bash
   # Research large codebases
   gemini -p "@src/ Analyze the architecture"

   # Research specific topics
   gemini -p "How to implement feature X in Google ADK?"
   ```

2. **Mobile Development**:
   - Screens in `apps/mobile/src/screens/`
   - Components in `apps/mobile/src/components/`
   - State in `apps/mobile/src/store/`

3. **AI Agent Development**:
   - Agents in `services/ai-agent/src/agents/`
   - Tools in `services/ai-agent/src/tools/`
   - Run tests with ADK evaluation datasets

## Next Steps

See [IMPLEMENTATION_PLAN.md](./IMPLEMENTATION_PLAN.md) for the detailed 12-week roadmap.

**Next priorities**:
1. ✅ ~~Implement photo picker with batch selection~~ **DONE**
2. ✅ ~~Build backend API for photo processing~~ **DONE**
3. Integrate Vision Agent with actual Gemini API calls
4. Create photo upload to Google Cloud Storage
5. Set up database (PostgreSQL/Supabase)
6. Connect mobile app to process photos with AI

## Resources

- [Google ADK Documentation](https://google.github.io/adk/)
- [Expo Documentation](https://docs.expo.dev/)
- [React Navigation](https://reactnavigation.org/)
- [Zustand](https://zustand-demo.pmnd.rs/)

## License

ISC
