export interface ThemeColors {
  background: { primary: string; surface: string; elevated: string };
  text: { primary: string; secondary: string; tertiary: string; inverse: string };
  border: { default: string; subtle: string };
  accent: { green: string; blue: string; purple: string; red: string; amber: string };
  accentTint: { green: string; blue: string; purple: string; red: string; amber: string };
  tabBar: { background: string; border: string; active: string; inactive: string };
  input: { background: string; border: string; placeholder: string };
  shimmer: { base: string; highlight: string };
  overlay: string;
}

export const lightColors: ThemeColors = {
  background: { primary: '#F5F5F5', surface: '#F3F4F6', elevated: '#FFFFFF' },
  text: { primary: '#111827', secondary: '#374151', tertiary: '#6B7280', inverse: '#FFFFFF' },
  border: { default: '#D1D5DB', subtle: '#E5E7EB' },
  accent: { green: '#16A34A', blue: '#3B82F6', purple: '#7C3AED', red: '#EF4444', amber: '#F59E0B' },
  accentTint: { green: '#F0FDF4', blue: '#EFF6FF', purple: '#F5F3FF', red: '#FEF2F2', amber: '#FFFBEB' },
  tabBar: { background: '#FFFFFF', border: '#E5E7EB', active: '#16A34A', inactive: '#9CA3AF' },
  input: { background: '#F3F4F6', border: '#D1D5DB', placeholder: '#9CA3AF' },
  shimmer: { base: '#E5E7EB', highlight: '#F3F4F6' },
  overlay: 'rgba(0,0,0,0.5)',
};

export const darkColors: ThemeColors = {
  background: { primary: '#0F172A', surface: '#1E293B', elevated: '#334155' },
  text: { primary: '#F9FAFB', secondary: '#D1D5DB', tertiary: '#9CA3AF', inverse: '#111827' },
  border: { default: '#374151', subtle: '#1E293B' },
  accent: { green: '#16A34A', blue: '#3B82F6', purple: '#7C3AED', red: '#EF4444', amber: '#F59E0B' },
  accentTint: { green: '#0D2818', blue: '#172033', purple: '#1A1528', red: '#2D1515', amber: '#2D2305' },
  tabBar: { background: '#1E293B', border: '#374151', active: '#16A34A', inactive: '#9CA3AF' },
  input: { background: '#334155', border: '#374151', placeholder: '#9CA3AF' },
  shimmer: { base: '#334155', highlight: '#475569' },
  overlay: 'rgba(0,0,0,0.7)',
};
