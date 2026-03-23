import { type Theme } from '@react-navigation/native';
import { lightColors, darkColors, type ThemeColors } from './colors';

function buildNavTheme(colors: ThemeColors, isDark: boolean): Theme {
  return {
    dark: isDark,
    colors: {
      primary: colors.accent.green,
      background: colors.background.primary,
      card: colors.background.elevated,
      text: colors.text.primary,
      border: colors.border.subtle,
      notification: colors.accent.red,
    },
    fonts: {
      regular: { fontFamily: 'System', fontWeight: '400' as const },
      medium: { fontFamily: 'System', fontWeight: '500' as const },
      bold: { fontFamily: 'System', fontWeight: '700' as const },
      heavy: { fontFamily: 'System', fontWeight: '900' as const },
    },
  };
}

export const lightNavTheme = buildNavTheme(lightColors, false);
export const darkNavTheme = buildNavTheme(darkColors, true);
