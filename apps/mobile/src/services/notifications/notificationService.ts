/**
 * Daily Macro Notification Service (NTF-01)
 *
 * Schedules a daily push notification summarizing the user's macro totals.
 * Uses expo-notifications with DailyTriggerInput for reliable daily delivery.
 *
 * Key exports:
 *   - requestNotificationPermission() — request/check notification permissions
 *   - scheduleDailyNotification(hour, minute, body) — schedule daily notification
 *   - cancelDailyNotification() — cancel all scheduled notifications
 *   - buildMacroSummaryBody(totals) — format macro totals for notification body
 *   - rescheduleWithFreshContent(hour, minute, getTotals) — rebuild body + reschedule
 */

import * as Notifications from 'expo-notifications';

/**
 * Request notification permissions.
 * Checks existing permissions first; only prompts if not already granted.
 *
 * @returns true if permissions are granted, false otherwise
 */
export async function requestNotificationPermission(): Promise<boolean> {
  const { status: existingStatus } = await Notifications.getPermissionsAsync();
  if (existingStatus === 'granted') return true;

  const { status } = await Notifications.requestPermissionsAsync();
  return status === 'granted';
}

/**
 * Schedule a daily notification at the specified time.
 * Cancels any existing scheduled notifications before scheduling the new one.
 *
 * @param hour - Hour of day (0-23)
 * @param minute - Minute of hour (0-59)
 * @param body - Notification body text (formatted macro summary)
 * @returns The notification identifier
 */
export async function scheduleDailyNotification(
  hour: number,
  minute: number,
  body: string,
): Promise<string> {
  // Cancel existing before scheduling new
  await Notifications.cancelAllScheduledNotificationsAsync();

  const id = await Notifications.scheduleNotificationAsync({
    content: {
      title: 'Daily Nutrition Summary',
      body,
      sound: true,
    },
    trigger: {
      type: 'daily' as any,
      hour,
      minute,
    },
  });

  return id;
}

/**
 * Cancel all scheduled notifications.
 */
export async function cancelDailyNotification(): Promise<void> {
  await Notifications.cancelAllScheduledNotificationsAsync();
}

/**
 * Format macro totals into a notification body string.
 *
 * @param totals - Today's macro totals
 * @returns Formatted string like "Cal: 1,850 | P: 120g | C: 200g | F: 65g"
 */
export function buildMacroSummaryBody(totals: {
  calories: number;
  protein: number;
  carbs: number;
  fat: number;
}): string {
  const cal = Math.round(totals.calories).toLocaleString('en-US');
  const p = Math.round(totals.protein);
  const c = Math.round(totals.carbs);
  const f = Math.round(totals.fat);
  return `Cal: ${cal} | P: ${p}g | C: ${c}g | F: ${f}g`;
}

/**
 * Reschedule the daily notification with fresh macro content.
 * Call this on app foreground to ensure the notification shows current totals.
 *
 * @param hour - Hour of day (0-23)
 * @param minute - Minute of hour (0-59)
 * @param getTotals - Function that returns today's macro totals
 */
export async function rescheduleWithFreshContent(
  hour: number,
  minute: number,
  getTotals: () => { calories: number; protein: number; carbs: number; fat: number },
): Promise<void> {
  const totals = getTotals();
  const body = buildMacroSummaryBody(totals);
  await scheduleDailyNotification(hour, minute, body);
}
