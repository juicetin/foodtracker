/**
 * driveAuth service tests — Google Sign-In wrapper with stale token retry.
 */

import {
  signInToGoogle,
  signOutGoogle,
  ensureDriveAccess,
  isSignedIn,
} from '../driveAuth';

// ---------------------------------------------------------------------------
// Mocks
// ---------------------------------------------------------------------------

const mockSignIn = jest.fn();
const mockSignOut = jest.fn();
const mockGetTokens = jest.fn();
const mockGetCurrentUser = jest.fn();
const mockHasPlayServices = jest.fn();
const mockClearCachedAccessToken = jest.fn();

jest.mock('@react-native-google-signin/google-signin', () => ({
  GoogleSignin: {
    configure: jest.fn(),
    hasPlayServices: (...a: unknown[]) => mockHasPlayServices(...a),
    signIn: (...a: unknown[]) => mockSignIn(...a),
    signOut: (...a: unknown[]) => mockSignOut(...a),
    getTokens: (...a: unknown[]) => mockGetTokens(...a),
    getCurrentUser: (...a: unknown[]) => mockGetCurrentUser(...a),
    clearCachedAccessToken: (...a: unknown[]) => mockClearCachedAccessToken(...a),
  },
  statusCodes: {
    SIGN_IN_CANCELLED: 'SIGN_IN_CANCELLED',
  },
}));

const mockSetProviderOptions = jest.fn();
jest.mock('react-native-cloud-storage', () => ({
  CloudStorage: {
    setProviderOptions: (...a: unknown[]) => mockSetProviderOptions(...a),
  },
}));

beforeEach(() => {
  jest.clearAllMocks();
});

// ---------------------------------------------------------------------------
// signInToGoogle
// ---------------------------------------------------------------------------

describe('signInToGoogle', () => {
  it('calls hasPlayServices + signIn and returns user info', async () => {
    const userInfo = { data: { user: { email: 'test@test.com', name: 'Test' } } };
    mockHasPlayServices.mockResolvedValue(true);
    mockSignIn.mockResolvedValue(userInfo);

    const result = await signInToGoogle();

    expect(mockHasPlayServices).toHaveBeenCalled();
    expect(mockSignIn).toHaveBeenCalled();
    expect(result).toEqual(userInfo);
  });

  it('throws descriptive error on SIGN_IN_CANCELLED', async () => {
    mockHasPlayServices.mockResolvedValue(true);
    const cancelError = new Error('cancelled');
    (cancelError as unknown as { code: string }).code = 'SIGN_IN_CANCELLED';
    mockSignIn.mockRejectedValue(cancelError);

    await expect(signInToGoogle()).rejects.toThrow('Sign-in was cancelled');
  });
});

// ---------------------------------------------------------------------------
// ensureDriveAccess
// ---------------------------------------------------------------------------

describe('ensureDriveAccess', () => {
  it('calls getTokens and sets accessToken on CloudStorage', async () => {
    mockGetTokens.mockResolvedValue({ accessToken: 'tok123' });

    await ensureDriveAccess();

    expect(mockGetTokens).toHaveBeenCalled();
    expect(mockSetProviderOptions).toHaveBeenCalledWith(
      expect.objectContaining({ accessToken: 'tok123' }),
    );
  });

  it('retries on 401 — clearCachedAccessToken + getTokens again', async () => {
    const err401 = new Error('401');
    (err401 as unknown as { code: number }).code = 401;
    mockGetTokens
      .mockRejectedValueOnce(err401)
      .mockResolvedValueOnce({ accessToken: 'newTok' });
    mockClearCachedAccessToken.mockResolvedValue(null);

    await ensureDriveAccess();

    expect(mockClearCachedAccessToken).toHaveBeenCalledWith('');
    expect(mockGetTokens).toHaveBeenCalledTimes(2);
    expect(mockSetProviderOptions).toHaveBeenCalledWith(
      expect.objectContaining({ accessToken: 'newTok' }),
    );
  });
});

// ---------------------------------------------------------------------------
// isSignedIn
// ---------------------------------------------------------------------------

describe('isSignedIn', () => {
  it('returns true when getCurrentUser returns user', () => {
    mockGetCurrentUser.mockReturnValue({ user: { email: 'a@b.c' } });
    expect(isSignedIn()).toBe(true);
  });

  it('returns false when getCurrentUser returns null', () => {
    mockGetCurrentUser.mockReturnValue(null);
    expect(isSignedIn()).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// signOutGoogle
// ---------------------------------------------------------------------------

describe('signOutGoogle', () => {
  it('calls GoogleSignin.signOut', async () => {
    mockSignOut.mockResolvedValue(null);
    await signOutGoogle();
    expect(mockSignOut).toHaveBeenCalled();
  });
});
