/**
 * Google Health Connect weight data import service.
 *
 * Wraps react-native-health-connect APIs with graceful error handling
 * for unsupported devices (Android < 14 without Health Connect app).
 */

import {
  initialize,
  getSdkStatus,
  requestPermission,
  readRecords,
  SdkAvailabilityStatus,
} from 'react-native-health-connect';

export interface HealthConnectWeightRecord {
  date: string; // YYYY-MM-DD
  weightKg: number;
  healthConnectId: string;
}

/**
 * Check if Health Connect is available on this device.
 * Returns false (no throw) on unsupported devices or when HC app is not installed.
 */
export async function isHealthConnectAvailable(): Promise<boolean> {
  try {
    const status = await getSdkStatus();
    return status === SdkAvailabilityStatus.SDK_AVAILABLE;
  } catch {
    return false;
  }
}

/** Initialize Health Connect SDK. Must be called before other operations. */
export async function initHealthConnect(): Promise<void> {
  await initialize();
}

/** Request read permission for weight records. Returns true if granted. */
export async function requestWeightPermission(): Promise<boolean> {
  const result = await requestPermission([
    { accessType: 'read', recordType: 'Weight' },
  ]);
  return result.length > 0 && result.some((r: any) => r.granted !== false);
}

/**
 * Read weight records from Health Connect within a date range.
 * Returns simplified objects with date (YYYY-MM-DD), weightKg, and healthConnectId.
 */
export async function readWeightRecords(
  startDate: Date,
  endDate: Date,
): Promise<HealthConnectWeightRecord[]> {
  const response = await readRecords('Weight', {
    timeRangeFilter: {
      operator: 'between',
      startTime: startDate.toISOString(),
      endTime: endDate.toISOString(),
    },
  });

  return (response.records ?? []).map((record: any) => ({
    date: record.time.split('T')[0],
    weightKg: record.weight.inKilograms,
    healthConnectId: record.metadata.id,
  }));
}
