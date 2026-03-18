import { requireNativeModule } from 'expo-modules-core';

const GeminiNanoNative = requireNativeModule('GeminiNano');

export type AvailabilityStatus = 'available' | 'downloading' | 'not_supported';

export const geminiNanoModule = {
  checkAvailability(): Promise<AvailabilityStatus> {
    return GeminiNanoNative.checkAvailability();
  },
  identifyFood(imageUri: string, prompt: string): Promise<string> {
    return GeminiNanoNative.identifyFood(imageUri, prompt);
  },
};
