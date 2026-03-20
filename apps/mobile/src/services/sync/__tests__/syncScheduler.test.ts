/**
 * syncScheduler tests -- background sync trigger with WiFi gate and Drive upload.
 */

import { triggerManualSync } from '../syncScheduler';

// ---------------------------------------------------------------------------
// Mocks
// ---------------------------------------------------------------------------

const mockIsSignedIn = jest.fn();
jest.mock('../driveAuth', () => ({
  isSignedIn: () => mockIsSignedIn(),
}));

const mockPerformIncrementalBackup = jest.fn();
jest.mock('../../backup/backupService', () => ({
  performIncrementalBackup: (...a: unknown[]) => mockPerformIncrementalBackup(...a),
}));

const mockUploadIncremental = jest.fn();
const mockDownloadSyncManifest = jest.fn();
const mockUploadSyncManifest = jest.fn();
jest.mock('../driveSync', () => ({
  uploadIncremental: (...a: unknown[]) => mockUploadIncremental(...a),
  downloadSyncManifest: (...a: unknown[]) => mockDownloadSyncManifest(...a),
  uploadSyncManifest: (...a: unknown[]) => mockUploadSyncManifest(...a),
}));

const mockNetInfoFetch = jest.fn();
jest.mock('@react-native-community/netinfo', () => ({
  __esModule: true,
  default: { fetch: () => mockNetInfoFetch() },
}));

// Mock useSyncStore as a simple object with getState/setState
const mockSyncState = {
  wifiOnly: true,
  syncStatus: 'idle' as string,
  lastSyncAt: null as string | null,
  setSyncStatus: jest.fn(),
  setLastSyncAt: jest.fn(),
};
jest.mock('../../../store/useSyncStore', () => ({
  useSyncStore: {
    getState: () => mockSyncState,
  },
}));

jest.mock('expo-file-system', () => ({
  Paths: { document: { uri: '/mock/docs/' } },
}));

beforeEach(() => {
  jest.clearAllMocks();
  mockSyncState.wifiOnly = true;
  mockSyncState.syncStatus = 'idle';
  mockSyncState.lastSyncAt = null;
});

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('triggerManualSync', () => {
  it('calls performIncrementalBackup, uploads result to Drive, updates sync manifest', async () => {
    mockIsSignedIn.mockReturnValue(true);
    mockNetInfoFetch.mockResolvedValue({ type: 'wifi', isConnected: true });
    mockPerformIncrementalBackup.mockResolvedValue({
      filename: 'backup-2026.json',
      changeCount: 5,
    });
    mockUploadIncremental.mockResolvedValue(undefined);
    mockDownloadSyncManifest.mockResolvedValue({
      deviceId: 'dev1',
      lastSyncedAt: '2026-03-20T10:00:00Z',
      lastFullBackupId: 'full-1',
      incrementalIds: ['inc-1'],
      appVersion: '1.0.0',
    });
    mockUploadSyncManifest.mockResolvedValue(undefined);

    await triggerManualSync();

    expect(mockPerformIncrementalBackup).toHaveBeenCalled();
    expect(mockUploadIncremental).toHaveBeenCalledWith(
      'backup-2026.json',
      expect.stringContaining('backup-2026.json'),
    );
    expect(mockUploadSyncManifest).toHaveBeenCalledWith(
      expect.objectContaining({
        incrementalIds: expect.arrayContaining(['inc-1', 'backup-2026.json']),
      }),
    );
    expect(mockSyncState.setLastSyncAt).toHaveBeenCalled();
  });

  it('skips upload when not signed in', async () => {
    mockIsSignedIn.mockReturnValue(false);

    await triggerManualSync();

    expect(mockPerformIncrementalBackup).not.toHaveBeenCalled();
    expect(mockUploadIncremental).not.toHaveBeenCalled();
  });

  it('respects WiFi-only setting (skips on cellular when wifiOnly=true)', async () => {
    mockIsSignedIn.mockReturnValue(true);
    mockSyncState.wifiOnly = true;
    mockNetInfoFetch.mockResolvedValue({ type: 'cellular', isConnected: true });

    await triggerManualSync();

    expect(mockPerformIncrementalBackup).not.toHaveBeenCalled();
  });

  it('allows sync on cellular when wifiOnly=false', async () => {
    mockIsSignedIn.mockReturnValue(true);
    mockSyncState.wifiOnly = false;
    mockNetInfoFetch.mockResolvedValue({ type: 'cellular', isConnected: true });
    mockPerformIncrementalBackup.mockResolvedValue(null);
    mockDownloadSyncManifest.mockResolvedValue(null);

    await triggerManualSync();

    expect(mockPerformIncrementalBackup).toHaveBeenCalled();
  });

  it('updates syncStore lastSyncAt on success', async () => {
    mockIsSignedIn.mockReturnValue(true);
    mockNetInfoFetch.mockResolvedValue({ type: 'wifi', isConnected: true });
    mockPerformIncrementalBackup.mockResolvedValue({
      filename: 'backup.json',
      changeCount: 1,
    });
    mockUploadIncremental.mockResolvedValue(undefined);
    mockDownloadSyncManifest.mockResolvedValue(null);
    mockUploadSyncManifest.mockResolvedValue(undefined);

    await triggerManualSync();

    expect(mockSyncState.setLastSyncAt).toHaveBeenCalledWith(expect.any(String));
    expect(mockSyncState.setSyncStatus).toHaveBeenCalledWith('idle');
  });

  it('sets syncStatus to error on Drive upload failure', async () => {
    mockIsSignedIn.mockReturnValue(true);
    mockNetInfoFetch.mockResolvedValue({ type: 'wifi', isConnected: true });
    mockPerformIncrementalBackup.mockResolvedValue({
      filename: 'backup.json',
      changeCount: 1,
    });
    mockUploadIncremental.mockRejectedValue(new Error('Drive upload failed'));

    await triggerManualSync();

    expect(mockSyncState.setSyncStatus).toHaveBeenCalledWith('error');
  });
});
