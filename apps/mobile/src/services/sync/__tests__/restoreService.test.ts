/**
 * restoreService tests -- discover and restore remote backups from Drive.
 */

import { discoverRemoteBackups, restoreFromDrive } from '../restoreService';

// ---------------------------------------------------------------------------
// Mocks
// ---------------------------------------------------------------------------

const mockDownloadSyncManifest = jest.fn();
const mockDownloadBackup = jest.fn();
jest.mock('../driveSync', () => ({
  downloadSyncManifest: (...a: unknown[]) => mockDownloadSyncManifest(...a),
  downloadBackup: (...a: unknown[]) => mockDownloadBackup(...a),
}));

jest.mock('../../../../db/client', () => ({
  opsqlite: {
    executeSync: jest.fn(),
    close: jest.fn(),
  },
}));

const mockOpen = jest.fn();
jest.mock('@op-engineering/op-sqlite', () => ({
  open: (...a: unknown[]) => mockOpen(...a),
}));

jest.mock('expo-file-system', () => ({
  Paths: { document: { uri: '/mock/docs/' } },
  File: jest.fn().mockImplementation((path: string) => ({
    exists: true,
    text: jest.fn().mockResolvedValue('{"changes":[]}'),
    uri: path,
    copy: jest.fn(),
    delete: jest.fn(),
  })),
}));

beforeEach(() => {
  jest.clearAllMocks();
});

// ---------------------------------------------------------------------------
// discoverRemoteBackups
// ---------------------------------------------------------------------------

describe('discoverRemoteBackups', () => {
  it('downloads sync manifest and returns it', async () => {
    const manifest = {
      deviceId: 'dev1',
      lastSyncedAt: '2026-03-20T10:00:00Z',
      lastFullBackupId: 'full-1',
      incrementalIds: ['inc-1', 'inc-2'],
      appVersion: '1.0.0',
    };
    mockDownloadSyncManifest.mockResolvedValue(manifest);

    const result = await discoverRemoteBackups();

    expect(mockDownloadSyncManifest).toHaveBeenCalled();
    expect(result).toEqual(manifest);
  });

  it('returns null when no manifest exists', async () => {
    mockDownloadSyncManifest.mockResolvedValue(null);

    const result = await discoverRemoteBackups();

    expect(result).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// restoreFromDrive
// ---------------------------------------------------------------------------

describe('restoreFromDrive', () => {
  const manifest = {
    deviceId: 'dev1',
    lastSyncedAt: '2026-03-20T12:00:00Z',
    lastFullBackupId: 'full-backup-2026.db',
    incrementalIds: ['inc-a.json', 'inc-b.json'],
    appVersion: '1.0.0',
  };

  it('downloads full backup first, then incrementals in order', async () => {
    mockDownloadSyncManifest.mockResolvedValue(manifest);
    mockDownloadBackup.mockResolvedValue('/local/path');
    const mockBackupDb = {
      executeSync: jest.fn(),
      close: jest.fn(),
    };
    mockOpen.mockReturnValue(mockBackupDb);

    await restoreFromDrive();

    // Full backup downloaded first
    expect(mockDownloadBackup).toHaveBeenNthCalledWith(
      1,
      'full-backup-2026.db',
      expect.any(String),
    );
    // Then incrementals in order
    expect(mockDownloadBackup).toHaveBeenNthCalledWith(
      2,
      'inc-a.json',
      expect.any(String),
    );
    expect(mockDownloadBackup).toHaveBeenNthCalledWith(
      3,
      'inc-b.json',
      expect.any(String),
    );
  });

  it('applies incrementals by timestamp order (oldest first)', async () => {
    mockDownloadSyncManifest.mockResolvedValue(manifest);
    mockDownloadBackup.mockResolvedValue('/local/path');
    const mockBackupDb = {
      executeSync: jest.fn(),
      close: jest.fn(),
    };
    mockOpen.mockReturnValue(mockBackupDb);

    await restoreFromDrive();

    // incrementalIds are in order already [inc-a, inc-b]
    // Both should be downloaded
    expect(mockDownloadBackup).toHaveBeenCalledTimes(3); // 1 full + 2 incrementals
  });

  it('throws if no manifest found', async () => {
    mockDownloadSyncManifest.mockResolvedValue(null);

    await expect(restoreFromDrive()).rejects.toThrow('No remote backups found');
  });
});
