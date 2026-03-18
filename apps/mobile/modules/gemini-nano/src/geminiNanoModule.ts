import { requireNativeModule } from 'expo-modules-core';

const GeminiNanoNative = requireNativeModule('GeminiNano');

export type AvailabilityStatus = 'available' | 'downloading' | 'downloadable' | 'unavailable' | 'needs_update';
export type DownloadResult = 'started' | 'already_available' | 'not_supported' | `error:${string}`;

export const geminiNanoModule = {
  checkAvailability(): Promise<AvailabilityStatus> {
    return GeminiNanoNative.checkAvailability();
  },
  requestDownload(): Promise<DownloadResult> {
    return GeminiNanoNative.requestDownload();
  },
  testTextOnly(prompt: string): Promise<string> {
    return GeminiNanoNative.testTextOnly(prompt);
  },
  identifyFood(imageUri: string, prompt: string): Promise<string> {
    return GeminiNanoNative.identifyFood(imageUri, prompt);
  },
};
