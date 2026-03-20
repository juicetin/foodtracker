/**
 * FTP credential storage and high-level upload/download operations.
 *
 * Credentials are stored securely via expo-secure-store (never AsyncStorage).
 * All FTP operations use passive mode via the native ftp-client module.
 */

import * as SecureStore from 'expo-secure-store';
import { ftpClientModule } from '../../../modules/ftp-client/src/ftpClientModule';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface FtpCredentials {
  host: string;
  port: number;
  username: string;
  password: string;
  remotePath: string;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const SECURE_STORE_KEY = 'ftp_credentials';

// ---------------------------------------------------------------------------
// Credential management (expo-secure-store)
// ---------------------------------------------------------------------------

/**
 * Save FTP credentials to secure store.
 */
export async function saveFtpCredentials(creds: FtpCredentials): Promise<void> {
  await SecureStore.setItemAsync(SECURE_STORE_KEY, JSON.stringify(creds));
}

/**
 * Load FTP credentials from secure store.
 * Returns null if no credentials are saved.
 */
export async function loadFtpCredentials(): Promise<FtpCredentials | null> {
  const json = await SecureStore.getItemAsync(SECURE_STORE_KEY);
  if (!json) return null;
  try {
    return JSON.parse(json) as FtpCredentials;
  } catch {
    return null;
  }
}

/**
 * Delete FTP credentials from secure store.
 */
export async function clearFtpCredentials(): Promise<void> {
  await SecureStore.deleteItemAsync(SECURE_STORE_KEY);
}

// ---------------------------------------------------------------------------
// FTP operations
// ---------------------------------------------------------------------------

/**
 * Upload a local file to the configured FTP server.
 * Loads credentials from secure store, constructs remote path.
 */
export async function uploadToFtp(
  localPath: string,
  remoteFilename: string,
): Promise<void> {
  const creds = await loadFtpCredentials();
  if (!creds) throw new Error('No FTP credentials configured');

  const remotePath = creds.remotePath.endsWith('/')
    ? `${creds.remotePath}${remoteFilename}`
    : `${creds.remotePath}/${remoteFilename}`;

  await ftpClientModule.upload(
    creds.host,
    creds.port,
    creds.username,
    creds.password,
    remotePath,
    localPath,
  );
}

/**
 * Test FTP connection using stored credentials.
 * Returns true if connection, login, and directory listing succeed.
 */
export async function testFtpConnection(): Promise<boolean> {
  const creds = await loadFtpCredentials();
  if (!creds) return false;

  return ftpClientModule.testConnection(
    creds.host,
    creds.port,
    creds.username,
    creds.password,
  );
}
