// Jest mock for the GeminiNano native module.
// Tests import from 'gemini-nano' and this file is resolved automatically by Jest
// because it lives in __mocks__ adjacent to src/.

export const geminiNanoModule = {
  checkAvailability: jest.fn().mockResolvedValue('not_supported'),
  requestDownload: jest.fn().mockResolvedValue('started'),
  identifyFood: jest.fn().mockResolvedValue(''),
};
