/**
 * syncScheduler tests -- multi-backend sync (Drive + FTP) with WiFi gate.
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

const mockSyncToFtp = jest.fn();
jest.mock('../ftpSync', () => ({
  syncToFtp: (...a: unknown[]) => mockSyncToFtp(...a),
}));

const mockLoadFtpCredentials = jest.fn();
jest.mock('../ftpClient', () => ({
  loadFtpCredentials: (...a: unknown[]) => mockLoadFtpCredentials(...a),
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
  ftpEnabled: false,
  ftpSyncStatus: 'idle' as string,
  lastFtpSyncAt: null as string | null,
  setSyncStatus: jest.fn(),
  setLastSyncAt: jest.fn(),
  setFtpSyncStatus: jest.fn(),
  setLastFtpSyncAt: jest.fn(),
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
  mockSyncState.ftpEnabled = false;
  mockSyncState.ftpSyncStatus = 'idle';
  mockSyncState.lastFtpSyncAt = null;
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

  it('still performs backup when not signed in (local backup still useful)', async () => {
    mockIsSignedIn.mockReturnValue(false);
    mockNetInfoFetch.mockResolvedValue({ type: 'wifi', isConnected: true });
    mockPerformIncrementalBackup.mockResolvedValue(null);

    await triggerManualSync();

    // Backup is always attempted; only Drive upload is skipped
    expect(mockPerformIncrementalBackup).toHaveBeenCalled();
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

  it('sets syncStatus to error on Drive upload failure (FTP unaffected)', async () => {
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

  // -------------------------------------------------------------------------
  // FTP tests
  // -------------------------------------------------------------------------

  it('dispatches to FTP when ftpEnabled and credentials present', async () => {
    mockIsSignedIn.mockReturnValue(false);
    mockSyncState.ftpEnabled = true;
    mockLoadFtpCredentials.mockResolvedValue({
      host: 'ftp.example.com',
      port: 21,
      username: 'user',
      password: 'pass',
      remotePath: '/backups',
    });
    mockNetInfoFetch.mockResolvedValue({ type: 'wifi', isConnected: true });
    mockPerformIncrementalBackup.mockResolvedValue({
      filename: 'backup.json',
      changeCount: 1,
    });
    mockSyncToFtp.mockResolvedValue(undefined);

    await triggerManualSync();

    expect(mockSyncToFtp).toHaveBeenCalledWith({ filename: 'backup.json', changeCount: 1 });
    expect(mockSyncState.setFtpSyncStatus).toHaveBeenCalledWith('idle');
    expect(mockSyncState.setLastFtpSyncAt).toHaveBeenCalledWith(expect.any(String));
  });

  it('dispatches to both Drive and FTP simultaneously via allSettled', async () => {
    mockIsSignedIn.mockReturnValue(true);
    mockSyncState.ftpEnabled = true;
    mockLoadFtpCredentials.mockResolvedValue({
      host: 'ftp.example.com',
      port: 21,
      username: 'user',
      password: 'pass',
      remotePath: '/backups',
    });
    mockNetInfoFetch.mockResolvedValue({ type: 'wifi', isConnected: true });
    mockPerformIncrementalBackup.mockResolvedValue({
      filename: 'backup.json',
      changeCount: 1,
    });
    mockUploadIncremental.mockResolvedValue(undefined);
    mockDownloadSyncManifest.mockResolvedValue(null);
    mockUploadSyncManifest.mockResolvedValue(undefined);
    mockSyncToFtp.mockResolvedValue(undefined);

    await triggerManualSync();

    // Both backends called
    expect(mockUploadIncremental).toHaveBeenCalled();
    expect(mockSyncToFtp).toHaveBeenCalled();
    expect(mockSyncState.setSyncStatus).toHaveBeenCalledWith('idle');
    expect(mockSyncState.setFtpSyncStatus).toHaveBeenCalledWith('idle');
  });

  it('FTP failure does not block Drive success', async () => {
    mockIsSignedIn.mockReturnValue(true);
    mockSyncState.ftpEnabled = true;
    mockLoadFtpCredentials.mockResolvedValue({
      host: 'ftp.example.com',
      port: 21,
      username: 'user',
      password: 'pass',
      remotePath: '/backups',
    });
    mockNetInfoFetch.mockResolvedValue({ type: 'wifi', isConnected: true });
    mockPerformIncrementalBackup.mockResolvedValue({
      filename: 'backup.json',
      changeCount: 1,
    });
    mockUploadIncremental.mockResolvedValue(undefined);
    mockDownloadSyncManifest.mockResolvedValue(null);
    mockUploadSyncManifest.mockResolvedValue(undefined);
    mockSyncToFtp.mockRejectedValue(new Error('FTP connection refused'));

    await triggerManualSync();

    // Drive succeeded, FTP errored
    expect(mockSyncState.setSyncStatus).toHaveBeenCalledWith('idle');
    expect(mockSyncState.setFtpSyncStatus).toHaveBeenCalledWith('error');
    expect(mockSyncState.setLastSyncAt).toHaveBeenCalled();
  });

  it('skips FTP when ftpEnabled but no credentials', async () => {
    mockIsSignedIn.mockReturnValue(true);
    mockSyncState.ftpEnabled = true;
    mockLoadFtpCredentials.mockResolvedValue(null);
    mockNetInfoFetch.mockResolvedValue({ type: 'wifi', isConnected: true });
    mockPerformIncrementalBackup.mockResolvedValue({
      filename: 'backup.json',
      changeCount: 1,
    });
    mockUploadIncremental.mockResolvedValue(undefined);
    mockDownloadSyncManifest.mockResolvedValue(null);
    mockUploadSyncManifest.mockResolvedValue(undefined);

    await triggerManualSync();

    expect(mockSyncToFtp).not.toHaveBeenCalled();
    expect(mockUploadIncremental).toHaveBeenCalled();
  });
});
