/**
 * Google Sign-In wrapper for Drive sync.
 *
 * Handles OAuth, token management, and scope escalation.
 * Implements Android stale-token retry pattern: on 401,
 * clear cached access token and retry once.
 */

import { GoogleSignin, statusCodes } from '@react-native-google-signin/google-signin';
import { CloudStorage } from 'react-native-cloud-storage';

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

// Configure Google Sign-In with drive.appdata scope.
// webClientId is set at runtime via configureGoogleSignIn() after
// the Google Cloud Console project is created.
let configured = false;

export function configureGoogleSignIn(webClientId?: string): void {
  GoogleSignin.configure({
    scopes: ['https://www.googleapis.com/auth/drive.appdata'],
    ...(webClientId ? { webClientId } : {}),
  });
  configured = true;
}

function ensureConfigured(): void {
  if (!configured) {
    configureGoogleSignIn();
  }
}

// ---------------------------------------------------------------------------
// Sign-in / sign-out
// ---------------------------------------------------------------------------

export async function signInToGoogle() {
  ensureConfigured();
  await GoogleSignin.hasPlayServices();
  try {
    const userInfo = await GoogleSignin.signIn();
    return userInfo;
  } catch (error: unknown) {
    const err = error as { code?: string; message?: string };
    if (err.code === statusCodes.SIGN_IN_CANCELLED) {
      throw new Error('Sign-in was cancelled');
    }
    throw error;
  }
}

export async function signOutGoogle(): Promise<void> {
  await GoogleSignin.signOut();
}

// ---------------------------------------------------------------------------
// Token management
// ---------------------------------------------------------------------------

export async function ensureDriveAccess(): Promise<void> {
  ensureConfigured();
  try {
    const tokens = await GoogleSignin.getTokens();
    CloudStorage.setProviderOptions({ accessToken: tokens.accessToken });
  } catch (error: unknown) {
    const err = error as { code?: number; status?: number };
    if (err.code === 401 || err.status === 401) {
      // Android stale token retry: clear cached token and try again
      await GoogleSignin.clearCachedAccessToken('');
      const tokens = await GoogleSignin.getTokens();
      CloudStorage.setProviderOptions({ accessToken: tokens.accessToken });
    } else {
      throw error;
    }
  }
}

// ---------------------------------------------------------------------------
// Scope escalation
// ---------------------------------------------------------------------------

export async function addDriveFileScope(): Promise<void> {
  ensureConfigured();
  await GoogleSignin.addScopes({
    scopes: ['https://www.googleapis.com/auth/drive.file'],
  });
}

// ---------------------------------------------------------------------------
// Status
// ---------------------------------------------------------------------------

export function isSignedIn(): boolean {
  return GoogleSignin.getCurrentUser() !== null;
}
