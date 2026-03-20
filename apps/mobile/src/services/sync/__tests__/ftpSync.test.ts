/**
 * ftpSync tests -- FTP sync operations mirroring driveSync pattern.
 */

import { syncToFtp } from '../ftpSync';

// ---------------------------------------------------------------------------
// Mocks
// ---------------------------------------------------------------------------

const mockUploadToFtp = jest.fn();
jest.mock('../ftpClient', () => ({
  uploadToFtp: (...a: unknown[]) => mockUploadToFtp(...a),
}));

jest.mock('expo-file-system', () => ({
  Paths: { document: { uri: '/mock/docs' } },
}));

beforeEach(() => {
  jest.clearAllMocks();
});

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('syncToFtp', () => {
  it('constructs correct local path and calls uploadToFtp', async () => {
    mockUploadToFtp.mockResolvedValue(undefined);

    await syncToFtp({ filename: 'backup-2026.json' });

    expect(mockUploadToFtp).toHaveBeenCalledWith(
      '/mock/docs/backups/backup-2026.json',
      'backup-2026.json',
    );
  });

  it('propagates upload errors', async () => {
    mockUploadToFtp.mockRejectedValue(new Error('FTP upload failed'));

    await expect(syncToFtp({ filename: 'backup.json' })).rejects.toThrow(
      'FTP upload failed',
    );
  });
});
