/**
 * Tests for the withAiPack Expo config plugin.
 *
 * Verifies that the plugin correctly modifies settings.gradle and
 * app/build.gradle to wire AI pack references for Play for On-Device AI.
 */

const withAiPack = require('../withAiPack');

// Mock @expo/config-plugins -- capture the modifier callbacks
const mockModifiers = {};

jest.mock('@expo/config-plugins', () => ({
  withSettingsGradle: (config, modifier) => {
    mockModifiers.settingsGradle = modifier;
    return config;
  },
  withAppBuildGradle: (config, modifier) => {
    mockModifiers.appBuildGradle = modifier;
    return config;
  },
}));

describe('withAiPack', () => {
  beforeEach(() => {
    Object.keys(mockModifiers).forEach((k) => delete mockModifiers[k]);
  });

  it('returns config unchanged when no packs provided', () => {
    const config = { name: 'test' };
    const result = withAiPack(config, { packs: [] });
    expect(result).toBe(config);
    expect(mockModifiers.settingsGradle).toBeUndefined();
  });

  it('adds include directive to settings.gradle for each pack', () => {
    const config = { name: 'test' };
    withAiPack(config, {
      packs: [{ name: 'ml-models', deliveryType: 'fast-follow' }],
    });

    expect(mockModifiers.settingsGradle).toBeDefined();

    const settingsConfig = {
      modResults: { contents: '// existing settings' },
    };
    const result = mockModifiers.settingsGradle(settingsConfig);

    expect(result.modResults.contents).toContain("include ':ml-models'");
    expect(result.modResults.contents).toContain(
      "project(':ml-models').projectDir = new File(rootProject.projectDir, '../ai-packs/ml-models')"
    );
  });

  it('does not duplicate include directive if already present', () => {
    const config = { name: 'test' };
    withAiPack(config, {
      packs: [{ name: 'ml-models', deliveryType: 'fast-follow' }],
    });

    const settingsConfig = {
      modResults: { contents: "include ':ml-models'\nproject(':ml-models').projectDir = new File(rootProject.projectDir, '../ai-packs/ml-models')" },
    };
    const result = mockModifiers.settingsGradle(settingsConfig);
    const matches = result.modResults.contents.match(/include ':ml-models'/g);
    expect(matches).toHaveLength(1);
  });

  it('adds assetPacks to app/build.gradle android block', () => {
    const config = { name: 'test' };
    withAiPack(config, {
      packs: [{ name: 'ml-models', deliveryType: 'fast-follow' }],
    });

    expect(mockModifiers.appBuildGradle).toBeDefined();

    const buildConfig = {
      modResults: {
        contents: 'android {\n    compileSdkVersion 35\n}',
      },
    };
    const result = mockModifiers.appBuildGradle(buildConfig);

    expect(result.modResults.contents).toContain("assetPacks += [':ml-models']");
  });

  it('does not duplicate assetPacks if already present', () => {
    const config = { name: 'test' };
    withAiPack(config, {
      packs: [{ name: 'ml-models', deliveryType: 'fast-follow' }],
    });

    const buildConfig = {
      modResults: {
        contents: "android {\n    assetPacks += [':ml-models']\n    compileSdkVersion 35\n}",
      },
    };
    const result = mockModifiers.appBuildGradle(buildConfig);

    const matches = result.modResults.contents.match(/assetPacks/g);
    expect(matches).toHaveLength(1);
  });

  it('handles multiple packs', () => {
    const config = { name: 'test' };
    withAiPack(config, {
      packs: [
        { name: 'ml-models', deliveryType: 'fast-follow' },
        { name: 'vlm-models', deliveryType: 'on-demand' },
      ],
    });

    const buildConfig = {
      modResults: {
        contents: 'android {\n    compileSdkVersion 35\n}',
      },
    };
    const result = mockModifiers.appBuildGradle(buildConfig);
    expect(result.modResults.contents).toContain("':ml-models', ':vlm-models'");
  });
});
