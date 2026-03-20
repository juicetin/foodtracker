/**
 * Tests for the ai-pack-delivery TypeScript module.
 *
 * Mocks the native module and verifies the TypeScript wrapper
 * correctly delegates to native methods with expected arguments.
 */

// Singleton mock native object -- returned by every requireNativeModule call.
// Must be defined inside jest.mock factory to avoid hoisting issues.
jest.mock('expo-modules-core', () => {
  const native = {
    getPackStatus: jest.fn().mockResolvedValue('completed'),
    getPackLocation: jest.fn().mockResolvedValue('/data/app/ai-packs/ml-models'),
    requestDownload: jest.fn().mockResolvedValue(true),
  };
  return {
    requireNativeModule: () => native,
    __mockNative: native,
  };
});

import { aiPackDeliveryModule } from '../aiPackDeliveryModule';
import type { AiPackStatus } from '../aiPackDeliveryModule';

// Access the shared mock for assertions
const { __mockNative: nativeMock } = require('expo-modules-core');

describe('aiPackDeliveryModule', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    // Restore defaults after clearAllMocks wipes implementation
    nativeMock.getPackStatus.mockResolvedValue('completed');
    nativeMock.getPackLocation.mockResolvedValue('/data/app/ai-packs/ml-models');
    nativeMock.requestDownload.mockResolvedValue(true);
  });

  it('getPackStatus returns status string', async () => {
    const status: AiPackStatus = await aiPackDeliveryModule.getPackStatus('ml-models');
    expect(status).toBe('completed');
    expect(nativeMock.getPackStatus).toHaveBeenCalledWith('ml-models');
  });

  it('getPackLocation returns path when completed', async () => {
    const path = await aiPackDeliveryModule.getPackLocation('ml-models');
    expect(path).toBe('/data/app/ai-packs/ml-models');
    expect(nativeMock.getPackLocation).toHaveBeenCalledWith('ml-models');
  });

  it('requestDownload returns boolean', async () => {
    const result = await aiPackDeliveryModule.requestDownload('ml-models');
    expect(result).toBe(true);
    expect(nativeMock.requestDownload).toHaveBeenCalledWith('ml-models');
  });

  it('getPackLocation returns null for unavailable pack', async () => {
    nativeMock.getPackLocation.mockResolvedValueOnce(null);
    const path = await aiPackDeliveryModule.getPackLocation('nonexistent');
    expect(path).toBeNull();
  });
});
