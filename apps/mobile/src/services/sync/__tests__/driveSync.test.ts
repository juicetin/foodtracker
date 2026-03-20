/**
 * driveSync service tests — Drive file operations via react-native-cloud-storage.
 */

import {
  uploadIncremental,
  uploadFullBackup,
  downloadBackup,
  listRemoteBackups,
  uploadSyncManifest,
} from '../driveSync';
import type { SyncManifest, RemoteBackupEntry } from '../types';

// ---------------------------------------------------------------------------
// Mocks
// ---------------------------------------------------------------------------

const mockEnsureDriveAccess = jest.fn().mockResolvedValue(undefined);
jest.mock('../driveAuth', () => ({
  ensureDriveAccess: (...a: unknown[]) => mockEnsureDriveAccess(...a),
}));

const mockWriteFile = jest.fn();
const mockUploadFile = jest.fn();
const mockDownloadFile = jest.fn();
const mockReadFile = jest.fn();

jest.mock('react-native-cloud-storage', () => ({
  CloudStorage: {
    writeFile: (...a: unknown[]) => mockWriteFile(...a),
    uploadFile: (...a: unknown[]) => mockUploadFile(...a),
    downloadFile: (...a: unknown[]) => mockDownloadFile(...a),
    readFile: (...a: unknown[]) => mockReadFile(...a),
  },
  CloudStorageScope: {
    AppData: 'app_data',
  },
}));

// Mock expo-file-system for reading local files
jest.mock('expo-file-system', () => ({
  Paths: { document: { uri: '/mock/docs/' } },
  File: jest.fn().mockImplementation((path: string) => ({
    exists: true,
    text: jest.fn().mockResolvedValue('{"test": true}'),
    uri: path,
  })),
}));

beforeEach(() => {
  jest.clearAllMocks();
});

// ---------------------------------------------------------------------------
// uploadIncremental
// ---------------------------------------------------------------------------

describe('uploadIncremental', () => {
  it('reads JSON file and calls CloudStorage.writeFile with AppData scope', async () => {
    mockWriteFile.mockResolvedValue(undefined);

    await uploadIncremental('backup-2026.json', '/local/path/backup-2026.json');

    expect(mockEnsureDriveAccess).toHaveBeenCalled();
    expect(mockWriteFile).toHaveBeenCalledWith(
      expect.stringContaining('backup-2026.json'),
      expect.any(String),
      'app_data',
    );
  });
});

// ---------------------------------------------------------------------------
// uploadFullBackup
// ---------------------------------------------------------------------------

describe('uploadFullBackup', () => {
  it('calls CloudStorage.uploadFile with sqlite3 mimeType and AppData scope', async () => {
    mockUploadFile.mockResolvedValue(undefined);

    await uploadFullBackup('full-backup.db', '/local/path/full-backup.db');

    expect(mockEnsureDriveAccess).toHaveBeenCalled();
    expect(mockUploadFile).toHaveBeenCalledWith(
      expect.stringContaining('full-backup.db'),
      '/local/path/full-backup.db',
      'app_data',
      expect.objectContaining({ mimeType: 'application/x-sqlite3' }),
    );
  });
});

// ---------------------------------------------------------------------------
// downloadBackup
// ---------------------------------------------------------------------------

describe('downloadBackup', () => {
  it('calls CloudStorage.downloadFile to local path', async () => {
    mockDownloadFile.mockResolvedValue('/local/dest/backup.db');

    const result = await downloadBackup('remote-backup.db', '/local/dest/backup.db');

    expect(mockEnsureDriveAccess).toHaveBeenCalled();
    expect(mockDownloadFile).toHaveBeenCalledWith(
      expect.stringContaining('remote-backup.db'),
      '/local/dest/backup.db',
      'app_data',
    );
    expect(result).toBe('/local/dest/backup.db');
  });
});

// ---------------------------------------------------------------------------
// listRemoteBackups
// ---------------------------------------------------------------------------

describe('listRemoteBackups', () => {
  it('returns RemoteBackupEntry[] derived from manifest', async () => {
    const manifest: SyncManifest = {
      deviceId: 'dev1',
      lastSyncedAt: '2026-03-20T10:00:00Z',
      lastFullBackupId: 'full-1',
      incrementalIds: ['inc-1', 'inc-2'],
      appVersion: '1.0.0',
    };
    mockReadFile.mockResolvedValue(JSON.stringify(manifest));

    const result = await listRemoteBackups();

    expect(mockEnsureDriveAccess).toHaveBeenCalled();
    expect(mockReadFile).toHaveBeenCalledWith(
      expect.stringContaining('sync-manifest.json'),
      'app_data',
    );
    expect(result).toEqual<RemoteBackupEntry[]>([
      { id: 'full-1', type: 'full', filename: 'full-1', uploadedAt: '2026-03-20T10:00:00Z', sizeBytes: null },
      { id: 'inc-1', type: 'incremental', filename: 'inc-1', uploadedAt: '2026-03-20T10:00:00Z', sizeBytes: null },
      { id: 'inc-2', type: 'incremental', filename: 'inc-2', uploadedAt: '2026-03-20T10:00:00Z', sizeBytes: null },
    ]);
  });

  it('returns empty array when no manifest exists', async () => {
    mockReadFile.mockRejectedValue(new Error('File not found'));

    const result = await listRemoteBackups();

    expect(result).toEqual([]);
  });
});

// ---------------------------------------------------------------------------
// uploadSyncManifest
// ---------------------------------------------------------------------------

describe('uploadSyncManifest', () => {
  it('writes manifest JSON to Drive appdata', async () => {
    const manifest: SyncManifest = {
      deviceId: 'dev1',
      lastSyncedAt: '2026-03-20T10:00:00Z',
      lastFullBackupId: null,
      incrementalIds: [],
      appVersion: '1.0.0',
    };
    mockWriteFile.mockResolvedValue(undefined);

    await uploadSyncManifest(manifest);

    expect(mockEnsureDriveAccess).toHaveBeenCalled();
    expect(mockWriteFile).toHaveBeenCalledWith(
      expect.stringContaining('sync-manifest.json'),
      JSON.stringify(manifest, null, 2),
      'app_data',
    );
  });
});

// ---------------------------------------------------------------------------
// All operations call ensureDriveAccess before Drive operation
// ---------------------------------------------------------------------------

describe('token-first pattern', () => {
  it('all operations call ensureDriveAccess first', async () => {
    mockWriteFile.mockResolvedValue(undefined);
    mockUploadFile.mockResolvedValue(undefined);
    mockDownloadFile.mockResolvedValue('/dest');
    mockReadFile.mockResolvedValue('{}');

    await uploadIncremental('a.json', '/p');
    await uploadFullBackup('b.db', '/p');
    await downloadBackup('c.db', '/d');
    await listRemoteBackups();
    await uploadSyncManifest({
      deviceId: 'x', lastSyncedAt: '', lastFullBackupId: null,
      incrementalIds: [], appVersion: '1.0.0',
    });

    expect(mockEnsureDriveAccess).toHaveBeenCalledTimes(5);
  });
});
