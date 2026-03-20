/**
 * ftpClient tests -- secure credential storage and FTP operations.
 */

import {
  saveFtpCredentials,
  loadFtpCredentials,
  clearFtpCredentials,
  uploadToFtp,
  testFtpConnection,
} from '../ftpClient';
import type { FtpCredentials } from '../ftpClient';

// ---------------------------------------------------------------------------
// Mocks
// ---------------------------------------------------------------------------

const mockSetItemAsync = jest.fn();
const mockGetItemAsync = jest.fn();
const mockDeleteItemAsync = jest.fn();
jest.mock('expo-secure-store', () => ({
  setItemAsync: (...a: unknown[]) => mockSetItemAsync(...a),
  getItemAsync: (...a: unknown[]) => mockGetItemAsync(...a),
  deleteItemAsync: (...a: unknown[]) => mockDeleteItemAsync(...a),
}));

const mockUpload = jest.fn();
const mockDownload = jest.fn();
const mockTestConnection = jest.fn();
jest.mock('../../../../modules/ftp-client/src/ftpClientModule', () => ({
  ftpClientModule: {
    upload: (...a: unknown[]) => mockUpload(...a),
    download: (...a: unknown[]) => mockDownload(...a),
    testConnection: (...a: unknown[]) => mockTestConnection(...a),
  },
}));

beforeEach(() => {
  jest.clearAllMocks();
});

// ---------------------------------------------------------------------------
// Credential tests
// ---------------------------------------------------------------------------

const testCreds: FtpCredentials = {
  host: 'ftp.example.com',
  port: 21,
  username: 'user',
  password: 's3cret',
  remotePath: '/backups',
};

describe('saveFtpCredentials', () => {
  it('stores credentials as JSON in SecureStore', async () => {
    await saveFtpCredentials(testCreds);
    expect(mockSetItemAsync).toHaveBeenCalledWith(
      'ftp_credentials',
      JSON.stringify(testCreds),
    );
  });
});

describe('loadFtpCredentials', () => {
  it('parses stored JSON credentials', async () => {
    mockGetItemAsync.mockResolvedValue(JSON.stringify(testCreds));
    const result = await loadFtpCredentials();
    expect(result).toEqual(testCreds);
  });

  it('returns null when no credentials stored', async () => {
    mockGetItemAsync.mockResolvedValue(null);
    const result = await loadFtpCredentials();
    expect(result).toBeNull();
  });

  it('returns null on invalid JSON', async () => {
    mockGetItemAsync.mockResolvedValue('not-json');
    const result = await loadFtpCredentials();
    expect(result).toBeNull();
  });
});

describe('clearFtpCredentials', () => {
  it('deletes credentials from SecureStore', async () => {
    await clearFtpCredentials();
    expect(mockDeleteItemAsync).toHaveBeenCalledWith('ftp_credentials');
  });
});

// ---------------------------------------------------------------------------
// Upload tests
// ---------------------------------------------------------------------------

describe('uploadToFtp', () => {
  it('loads credentials and calls native upload with correct remote path', async () => {
    mockGetItemAsync.mockResolvedValue(JSON.stringify(testCreds));
    mockUpload.mockResolvedValue(undefined);

    await uploadToFtp('/local/backup.json', 'backup.json');

    expect(mockUpload).toHaveBeenCalledWith(
      'ftp.example.com',
      21,
      'user',
      's3cret',
      '/backups/backup.json',
      '/local/backup.json',
    );
  });

  it('handles remotePath with trailing slash', async () => {
    const credsWithSlash = { ...testCreds, remotePath: '/backups/' };
    mockGetItemAsync.mockResolvedValue(JSON.stringify(credsWithSlash));
    mockUpload.mockResolvedValue(undefined);

    await uploadToFtp('/local/backup.json', 'backup.json');

    expect(mockUpload).toHaveBeenCalledWith(
      'ftp.example.com',
      21,
      'user',
      's3cret',
      '/backups/backup.json',
      '/local/backup.json',
    );
  });

  it('throws when no credentials configured', async () => {
    mockGetItemAsync.mockResolvedValue(null);
    await expect(uploadToFtp('/local/backup.json', 'backup.json')).rejects.toThrow(
      'No FTP credentials configured',
    );
  });
});

// ---------------------------------------------------------------------------
// Test connection tests
// ---------------------------------------------------------------------------

describe('testFtpConnection', () => {
  it('returns true when native testConnection succeeds', async () => {
    mockGetItemAsync.mockResolvedValue(JSON.stringify(testCreds));
    mockTestConnection.mockResolvedValue(true);

    const result = await testFtpConnection();
    expect(result).toBe(true);
    expect(mockTestConnection).toHaveBeenCalledWith(
      'ftp.example.com',
      21,
      'user',
      's3cret',
    );
  });

  it('returns false when no credentials stored', async () => {
    mockGetItemAsync.mockResolvedValue(null);
    const result = await testFtpConnection();
    expect(result).toBe(false);
  });

  it('returns false when native testConnection fails', async () => {
    mockGetItemAsync.mockResolvedValue(JSON.stringify(testCreds));
    mockTestConnection.mockResolvedValue(false);

    const result = await testFtpConnection();
    expect(result).toBe(false);
  });
});
