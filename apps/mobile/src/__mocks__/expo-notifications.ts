/**
 * Jest mock for expo-notifications.
 *
 * Provides jest.fn() stubs for the notification APIs used by notificationService.
 */

export const scheduleNotificationAsync = jest.fn().mockResolvedValue('mock-notification-id');
export const cancelAllScheduledNotificationsAsync = jest.fn().mockResolvedValue(undefined);
export const getPermissionsAsync = jest.fn().mockResolvedValue({ status: 'granted' });
export const requestPermissionsAsync = jest.fn().mockResolvedValue({ status: 'granted' });
export const setNotificationHandler = jest.fn();

export default {
  scheduleNotificationAsync,
  cancelAllScheduledNotificationsAsync,
  getPermissionsAsync,
  requestPermissionsAsync,
  setNotificationHandler,
};
