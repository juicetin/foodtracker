# External Integrations

**Analysis Date:** 2026-02-12

## APIs & External Services

**Google AI Services:**
- Gemini LLM - Food image analysis and ingredient identification
  - SDK/Client: `@google/adk` (Google Agent Development Kit)
  - Auth: Via Google Cloud credentials (configured in environment)
  - Usage: `services/ai-agent/` orchestrates multi-agent workflows using Gemini
  - Agents: VisionAgent, ScaleAgent, CoordinatorAgent

**Nutrition Data APIs:**
- USDA FoodData Central - Nutritional information lookup
  - SDK/Client: `requests` library (Python spike) / Node.js HTTP calls
  - Auth: DEMO_KEY available for testing (mentioned in CLAUDE.md)
  - Endpoint: `https://api.nal.usda.gov/fdc/v1`
  - Usage: `services/ai-agent/src/config/databases.ts` - Priority 1 for US region

- Open Food Facts - Fallback global nutrition database
  - Endpoint: `https://world.openfoodfacts.org/api/v0`
  - Auth: No authentication required (public API)
  - Usage: Lowest priority fallback in database lookup

**Regional Nutrition Databases (Offline/Local):**
- AFCD (Australian Food Composition Database) - AU region, Priority 1
- NUTTAB - AU region, Priority 2
- Canadian Nutrient File (CNF) - CA region
- Composition of Foods Integrated Dataset (CoFID) - UK region
- CIQUAL - FR region (France)

## Data Storage

**Databases:**
- PostgreSQL (production)
  - Connection: `process.env.DATABASE_URL`
  - Client: `pg` (Node.js driver)
  - Location: `backend/src/db/client.ts`
  - Features:
    - Connection pooling (max 20, 30s idle timeout)
    - Transaction support via `transaction()` helper
    - Query logging with performance metrics

**File Storage:**
- Local filesystem only (development)
- Expo File System API - Access to device storage (mobile)
  - Package: `expo-file-system` (19.0+)
  - Usage: Photo storage on device

**Caching:**
- None currently configured
- AsyncStorage available on mobile (`@react-native-async-storage/async-storage`) for offline state

## Authentication & Identity

**Auth Provider:**
- Custom implementation (not yet fully built)
- Client: `apps/mobile/src/lib/api/client.ts` supports Bearer token auth
- Token storage: Ready to use (structure in place, not yet populated)
- Flow: Bearer token in Authorization header

**Current Implementation:**
```typescript
// Mobile client supports auth tokens
if (this.token) {
  headers['Authorization'] = `Bearer ${this.token}`;
}
```

**Future Integration Points:**
- Auth provider not yet selected/implemented
- Structure exists for token-based authentication

## Monitoring & Observability

**Error Tracking:**
- None configured (not detected)

**Logs:**
- Console logging only
  - Backend: `console.log()` and `console.error()` in Express middleware
  - AI Service: Logging in agent execution
  - Database: Query execution logged with duration and row count
  - Example: `backend/src/index.ts` logs health check URL at startup

**No structured logging** (Pino, Winston, etc.) implemented yet

## CI/CD & Deployment

**Hosting:**
- Not configured yet
- Planned: EAS (Expo Application Services) for mobile builds
  - Mentioned in app.json and CLAUDE.md context

**CI Pipeline:**
- None detected
- GitHub Actions structure not found

**Backend Deployment:**
- Docker: No Dockerfile detected
- Platform: Node.js application ready to deploy (ExpressJS)
- Database migrations: Not detected (schema assumed to exist)

## Environment Configuration

**Required environment variables:**

| Variable | Service | Purpose |
|----------|---------|---------|
| `DATABASE_URL` | Backend | PostgreSQL connection string |
| `PORT` | Backend | Server port (default: 3100) |
| `GOOGLE_*` | AI Agent | Google Cloud credentials (exact names in ADK config) |
| API_BASE_URL (dev) | Mobile | Backend endpoint (hardcoded: http://localhost:3100/api) |

**Development Configuration:**
- Mobile API base: `http://localhost:3100/api` (local dev)
- Production API base: `https://api.foodtracker.com` (TODO: update before deployment)
- Location: `apps/mobile/src/lib/api/client.ts`

**Secrets location:**
- `.env` files (backend, mobile, services) - NOT committed to git
- Environment variables injected at runtime
- No secrets manager detected (AWS Secrets Manager, HashiCorp Vault, etc.)

## Webhooks & Callbacks

**Incoming:**
- None detected

**Outgoing:**
- None detected

## Regional Food Database Configuration

The system intelligently routes nutrition requests by user region:

```typescript
// Location: services/ai-agent/src/config/databases.ts
getDatabaseForRegion(region: string): FoodDatabase[]
```

**Resolution Order:**
1. User's region-specific database (highest priority)
2. Global fallback (Open Food Facts)

**Supported Regions:**
- AU → AFCD (1), NUTTAB (2), OpenFoodFacts (10)
- US → USDA FoodData Central (1), OpenFoodFacts (10)
- CA → Canadian Nutrient File (1), OpenFoodFacts (10)
- UK → CoFID (1), OpenFoodFacts (10)
- FR → CIQUAL (1), OpenFoodFacts (10)
- Global → OpenFoodFacts (10)

## Photo Processing Pipeline

**Mobile to Backend Flow:**

1. **Photo Capture** - `apps/mobile/src/components/PhotoPicker.tsx`
   - Uses `expo-image-picker`
   - Stores locally via `expo-file-system`

2. **Upload** - `apps/mobile/src/lib/api/client.ts`
   - Multipart/form-data via `FormData`
   - Endpoint: `/api/food-logs` (POST)

3. **Backend Processing** - `backend/src/index.ts`
   - multer middleware handles file upload
   - Stores reference in PostgreSQL
   - Sends to AI agent for analysis (planned)

4. **AI Analysis** - `services/ai-agent/src/agents/coordinatorAgent.ts`
   - Vision detection (image → food items)
   - Scale reading (if weight scale in image)
   - Volume estimation (3D reconstruction)
   - Nutrition lookup (regional database)

## Current Integration Status

**Production-Ready:**
- PostgreSQL database connectivity
- Express API framework
- Expo mobile platform
- Google ADK integration (SDK installed)

**In Development:**
- Full AI agent workflow (structure exists, mock implementations used)
- Authentication system (structure exists, not wired)
- Regional nutrition database routing (structure exists, mock data used)

**Not Started:**
- Error tracking service
- Structured logging
- CI/CD pipeline
- Cloud file storage (GCS, S3, etc.)
- Auth provider integration

---

*Integration audit: 2026-02-12*
