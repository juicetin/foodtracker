/**
 * Tests for notificationService — daily macro notification scheduling (NTF-01).
 */

import {
  scheduleDailyNotification,
  cancelDailyNotification,
  buildMacroSummaryBody,
} from '../notificationService';
import {
  scheduleNotificationAsync,
  cancelAllScheduledNotificationsAsync,
} from 'expo-notifications';

// ── Tests ──

beforeEach(() => {
  jest.clearAllMocks();
});

describe('scheduleDailyNotification', () => {
  it('calls scheduleNotificationAsync with DailyTriggerInput at specified hour/minute', async () => {
    await scheduleDailyNotification(21, 0, 'Cal: 1,850 | P: 120g | C: 200g | F: 65g');

    expect(scheduleNotificationAsync).toHaveBeenCalledTimes(1);
    const call = (scheduleNotificationAsync as jest.Mock).mock.calls[0][0];
    expect(call.content.title).toBe('Daily Nutrition Summary');
    expect(call.content.body).toBe('Cal: 1,850 | P: 120g | C: 200g | F: 65g');
    expect(call.trigger).toEqual(
      expect.objectContaining({
        type: 'daily',
        hour: 21,
        minute: 0,
      }),
    );
  });

  it('cancels existing notification before scheduling new one', async () => {
    await scheduleDailyNotification(20, 30, 'test body');

    // cancelAllScheduledNotificationsAsync should be called BEFORE scheduleNotificationAsync
    const cancelOrder = (cancelAllScheduledNotificationsAsync as jest.Mock).mock.invocationCallOrder[0];
    const scheduleOrder = (scheduleNotificationAsync as jest.Mock).mock.invocationCallOrder[0];
    expect(cancelOrder).toBeLessThan(scheduleOrder);
  });
});

describe('cancelDailyNotification', () => {
  it('calls cancelAllScheduledNotificationsAsync', async () => {
    await cancelDailyNotification();

    expect(cancelAllScheduledNotificationsAsync).toHaveBeenCalledTimes(1);
  });
});

describe('buildMacroSummaryBody', () => {
  it('formats totals as "Cal: 1,850 | P: 120g | C: 200g | F: 65g"', () => {
    const result = buildMacroSummaryBody({
      calories: 1850,
      protein: 120,
      carbs: 200,
      fat: 65,
    });

    expect(result).toBe('Cal: 1,850 | P: 120g | C: 200g | F: 65g');
  });

  it('formats zero totals correctly', () => {
    const result = buildMacroSummaryBody({
      calories: 0,
      protein: 0,
      carbs: 0,
      fat: 0,
    });

    expect(result).toBe('Cal: 0 | P: 0g | C: 0g | F: 0g');
  });
});
