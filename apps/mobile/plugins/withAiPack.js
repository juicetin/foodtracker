/**
 * Expo config plugin for Play for On-Device AI model delivery.
 *
 * Wires AI pack references into settings.gradle and app/build.gradle so that
 * the Android build system includes AI packs for Play Store delivery.
 *
 * NOTE: Requires AGP 8.8+ for com.android.ai-pack plugin support.
 * The actual AI pack build.gradle files live in ai-packs/<name>/build.gradle.
 */
const {
  withSettingsGradle,
  withAppBuildGradle,
} = require('@expo/config-plugins');

/**
 * @param {import('@expo/config-plugins').ExportedConfig} config
 * @param {{ packs: Array<{ name: string; deliveryType: 'fast-follow' | 'on-demand' }> }} props
 */
function withAiPack(config, props) {
  const { packs = [] } = props || {};

  if (packs.length === 0) return config;

  // 1. Add include directives to settings.gradle for each AI pack
  config = withSettingsGradle(config, (settingsConfig) => {
    const contents = settingsConfig.modResults.contents;
    for (const pack of packs) {
      const includeDirective = `include ':${pack.name}'`;
      if (!contents.includes(includeDirective)) {
        settingsConfig.modResults.contents =
          contents + `\n${includeDirective}\nproject(':${pack.name}').projectDir = new File(rootProject.projectDir, '../ai-packs/${pack.name}')\n`;
      }
    }
    return settingsConfig;
  });

  // 2. Add assetPacks references to app/build.gradle android block
  config = withAppBuildGradle(config, (buildConfig) => {
    const contents = buildConfig.modResults.contents;
    const packRefs = packs.map((p) => `':${p.name}'`).join(', ');
    const assetPacksLine = `    assetPacks += [${packRefs}]`;

    if (!contents.includes('assetPacks')) {
      // Insert assetPacks into the android { } block
      buildConfig.modResults.contents = contents.replace(
        /android\s*\{/,
        `android {\n${assetPacksLine}`
      );
    }
    return buildConfig;
  });

  return config;
}

module.exports = withAiPack;
