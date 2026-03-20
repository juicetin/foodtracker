/**
 * Mock for react-native-health-connect
 * Provides jest.fn() stubs for Health Connect API interactions.
 */

export const SdkAvailabilityStatus = {
  SDK_AVAILABLE: 3,
  SDK_UNAVAILABLE: 1,
  SDK_UNAVAILABLE_PROVIDER_UPDATE_REQUIRED: 2,
} as const;

export const initialize = jest.fn().mockResolvedValue(true);

export const getSdkStatus = jest
  .fn()
  .mockResolvedValue(SdkAvailabilityStatus.SDK_AVAILABLE);

export const requestPermission = jest.fn().mockResolvedValue([
  { accessType: 'read', recordType: 'Weight', granted: true },
]);

export const readRecords = jest.fn().mockResolvedValue({ records: [] });
