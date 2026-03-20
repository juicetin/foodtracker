import {
  getSdkStatus,
  initialize,
  requestPermission,
  readRecords,
  SdkAvailabilityStatus,
} from 'react-native-health-connect';
import {
  isHealthConnectAvailable,
  initHealthConnect,
  requestWeightPermission,
  readWeightRecords,
} from '../healthConnectService';

jest.mock('react-native-health-connect');

const mockGetSdkStatus = getSdkStatus as jest.Mock;
const mockInitialize = initialize as jest.Mock;
const mockRequestPermission = requestPermission as jest.Mock;
const mockReadRecords = readRecords as jest.Mock;

beforeEach(() => {
  jest.clearAllMocks();
});

describe('healthConnectService', () => {
  describe('isHealthConnectAvailable', () => {
    it('returns true when getSdkStatus returns SDK_AVAILABLE', async () => {
      mockGetSdkStatus.mockResolvedValue(SdkAvailabilityStatus.SDK_AVAILABLE);
      const result = await isHealthConnectAvailable();
      expect(result).toBe(true);
    });

    it('returns false and does not throw when Health Connect is not installed', async () => {
      mockGetSdkStatus.mockRejectedValue(new Error('Health Connect not installed'));
      const result = await isHealthConnectAvailable();
      expect(result).toBe(false);
    });

    it('returns false when SDK status is SDK_UNAVAILABLE', async () => {
      mockGetSdkStatus.mockResolvedValue(SdkAvailabilityStatus.SDK_UNAVAILABLE);
      const result = await isHealthConnectAvailable();
      expect(result).toBe(false);
    });
  });

  describe('initHealthConnect', () => {
    it('calls initialize()', async () => {
      await initHealthConnect();
      expect(mockInitialize).toHaveBeenCalledTimes(1);
    });
  });

  describe('requestWeightPermission', () => {
    it('returns true when permission granted', async () => {
      mockRequestPermission.mockResolvedValue([
        { accessType: 'read', recordType: 'Weight', granted: true },
      ]);
      const result = await requestWeightPermission();
      expect(result).toBe(true);
    });

    it('returns false when permission denied', async () => {
      mockRequestPermission.mockResolvedValue([]);
      const result = await requestWeightPermission();
      expect(result).toBe(false);
    });
  });

  describe('readWeightRecords', () => {
    it('returns array of { date, weightKg, healthConnectId } from Health Connect', async () => {
      mockReadRecords.mockResolvedValue({
        records: [
          {
            metadata: { id: 'hc-1' },
            time: '2025-01-15T08:00:00.000Z',
            weight: { inKilograms: 80.5 },
          },
          {
            metadata: { id: 'hc-2' },
            time: '2025-01-16T08:30:00.000Z',
            weight: { inKilograms: 80.2 },
          },
        ],
      });

      const start = new Date('2025-01-15');
      const end = new Date('2025-01-17');
      const result = await readWeightRecords(start, end);

      expect(result).toEqual([
        { date: '2025-01-15', weightKg: 80.5, healthConnectId: 'hc-1' },
        { date: '2025-01-16', weightKg: 80.2, healthConnectId: 'hc-2' },
      ]);
      expect(mockReadRecords).toHaveBeenCalledWith('Weight', {
        timeRangeFilter: {
          operator: 'between',
          startTime: start.toISOString(),
          endTime: end.toISOString(),
        },
      });
    });

    it('returns empty array when no records', async () => {
      mockReadRecords.mockResolvedValue({ records: [] });
      const result = await readWeightRecords(new Date(), new Date());
      expect(result).toEqual([]);
    });
  });
});
