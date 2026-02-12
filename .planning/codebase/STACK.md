# Technology Stack

**Analysis Date:** 2026-02-12

## Languages

**Primary:**
- TypeScript 5.x - Backend API, mobile app, AI agent service
- JavaScript/React Native - Mobile UI (React Native 0.81)
- Python 3.9 - ML/food detection POC only (spike directory, not production)

**Secondary:**
- Shell/Bash - Build scripts and CLI tools

## Runtime

**Environment:**
- Node.js 20.x (implied by package versions)
- Expo SDK 54 - Mobile app runtime

**Package Manager:**
- npm (lockfile: package-lock.json present in all packages)
- pip (Python dependencies in `spike/food-detection-poc/requirements.txt` only)

## Frameworks

**Core:**
- Express 4.18 - REST API framework (`backend/`)
- React Native 0.81 - Mobile application framework (`apps/mobile/`)
- Expo 54 - Managed React Native platform and build tooling (`apps/mobile/`)

**AI/ML:**
- @google/adk (latest) - Google Agent Development Kit for orchestrated AI workflows (`services/ai-agent/`)
- Gemini - LLM backend via Google ADK (configured in `services/ai-agent/src/agents/`)
- YOLOv8/v11 (ultralytics) - Object detection for food identification (spike POC only)

**State Management:**
- Zustand 5.0 - Lightweight state management for mobile app (`apps/mobile/package.json`)

**Navigation:**
- @react-navigation/native 7.1 - Core navigation for React Native
- @react-navigation/bottom-tabs 7.10 - Bottom tab navigation
- @react-navigation/native-stack 7.11 - Stack navigation

**UI/Visual:**
- @gorhom/bottom-sheet 5.2 - Bottom sheet component for mobile
- react-native-reanimated 4.2 - Animation library
- react-native-gesture-handler 2.30 - Gesture handling

## Testing

**Framework:**
- Jest 30.2 - Test runner for backend
- ts-jest 29.4 - TypeScript support for Jest
- @testing-library/react-native 13.3 - React Native testing utilities
- jest-expo 54 - Expo/React Native test configuration
- Supertest 7.2 - HTTP assertion library for API testing

**Run Commands:**
```bash
npm run test              # Run all tests
npm run test:watch       # Watch mode (backend)
NODE_OPTIONS=--experimental-vm-modules jest  # Run with ES modules
```

## Build & Development

**Build:**
- TypeScript 5.4+ - Transpilation (`tsc` command)
- tsx 4.7 - TypeScript execution for development (`tsx watch`)
- Expo CLI 54 - Build/run mobile apps

**Development Servers:**
- Express (dev): `npm run dev` via `tsx watch`
- Mobile: `expo start` (iOS/Android/Web)

## Key Dependencies

**Critical:**

| Package | Version | Purpose |
|---------|---------|---------|
| `pg` | 8.18 | PostgreSQL database driver |
| `@google/adk` | latest | Google Agent Development Kit - orchestrates food analysis agents |
| `express` | 4.18 | HTTP server and routing |
| `react-native` | 0.81 | Mobile application framework |
| `expo` | 54.0 | Managed React Native with EAS Build support |
| `dotenv` | 16.4 | Environment variable management |

**Infrastructure:**

| Package | Version | Purpose |
|---------|---------|---------|
| `cors` | 2.8 | Cross-Origin Resource Sharing middleware |
| `multer` | 1.4 | File upload handling (photos) |
| `@react-native-async-storage/async-storage` | 2.2 | Local persistent storage (mobile) |
| `expo-file-system` | 19.0 | File system access (photos, uploads) |
| `expo-image-picker` | 17.0 | Camera/gallery picker |
| `expo-media-library` | 18.2 | Access device media library |

**Data Handling (Spike/POC):**

| Package | Version | Purpose |
|---------|---------|---------|
| `ultralytics` (Python) | 8.3+ | YOLOv8/v11 food detection |
| `torch` (Python) | 2.0+ | PyTorch backend |
| `torchvision` (Python) | 0.15+ | Vision transforms and pretrained models |
| `opencv-python` | 4.8+ | Image processing and annotation |
| `requests` (Python) | 2.31+ | USDA FoodData Central API calls |

## Configuration Files

**TypeScript:**
- `backend/tsconfig.json` - Target ES2022, ESNext modules, strict mode
- `apps/mobile/tsconfig.json` - Extends expo/tsconfig.base with strict mode
- `services/ai-agent/tsconfig.json` - Same as backend (ES2022, ESNext)

**Testing:**
- `backend/jest.config.js` - ts-jest preset, ESM support, test pattern `**/__tests__/**/*.test.ts`

**Build/Runtime:**
- `apps/mobile/app.json` - Expo configuration (icon, splash, iOS/Android native config)

**Environment:**
- `.env` files are present but NOT committed to git (backend, mobile, services)
- Configuration via `process.env.DATABASE_URL` and `process.env.PORT`

## Database

**Production:**
- PostgreSQL (via `pg` driver)
- Connection: `process.env.DATABASE_URL`
- Pool: max 20 connections, 30s idle timeout, 2s connection timeout
- Location: `backend/src/db/client.ts`

## Platform Requirements

**Development:**
- Node.js 20+
- npm or npm-compatible package manager
- Xcode (for iOS development)
- Android Studio (for Android development)
- Python 3.9+ (for spike POC only)

**Production:**
- Node.js 20+ runtime for backend
- PostgreSQL 12+ database
- EAS (Expo Application Services) for mobile builds and over-the-air updates
- Google Cloud project with Gemini API access (for AI agents)

## API Endpoints (Backend)

**Health/Status:**
- `GET /health` - Service status check

**Food Tracking:**
- `POST/GET /api/food-logs` - Create/retrieve food entries
- `POST/GET /api/recipes` - Recipe management

**File Upload:**
- Supports multipart/form-data for photo uploads via multer

## Regional Support

The system supports multi-region food composition databases:
- AU (Australia) - AFCD, NUTTAB priority
- US (USA) - USDA FoodData Central priority
- CA (Canada) - Canadian Nutrient File
- UK - Composition of Foods Integrated Dataset (CoFID)
- FR (France) - CIQUAL
- Global fallback - Open Food Facts

---

*Stack analysis: 2026-02-12*
