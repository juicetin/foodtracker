/** @type {import('jest').Config} */
module.exports = {
  preset: 'jest-expo',
  transformIgnorePatterns: [
    'node_modules/(?!((jest-)?react-native|@react-native(-community)?)|expo(nent)?|@expo(nent)?/.*|@expo-google-fonts/.*|react-navigation|@react-navigation/.*|@sentry/react-native|native-base|react-native-svg|zustand|drizzle-orm)',
  ],
  moduleNameMapper: {
    '^@op-engineering/op-sqlite$':
      '<rootDir>/__mocks__/@op-engineering/op-sqlite.ts',
    '\\.tflite$': '<rootDir>/__mocks__/tfliteAsset.js',
    '^llama\\.rn$': '<rootDir>/__mocks__/llama.rn.ts',
  },
  testPathIgnorePatterns: ['/node_modules/', '/android/', '/ios/'],
};
